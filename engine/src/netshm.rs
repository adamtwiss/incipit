//! One copy of the network's big tables per host, shared by every engine
//! process using the same weights (the dev and base binaries of a test
//! included), held in 2 MB pages where the host has a hugepage pool.
//!
//! The copy lives in a memfd: hugetlb-backed when the pool can hold it,
//! ordinary shared memory otherwise. Processes find a live copy through a
//! registry file in /dev/shm named by a hash of the bytes, which lists the
//! holders' "pid fd" pairs; a newcomer opens /proc/<pid>/fd/<fd> of any
//! live holder and lists itself in turn. The memfd is freed by the kernel when its last holder
//! exits, so nothing outlives the engines but the tiny registry file.
//!
//! INCIPIT_NETSHM=0 disables this, =4k skips hugepages (for measurement).

#[cfg(target_os = "linux")]
mod imp {
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};

    extern "C" {
        fn memfd_create(name: *const u8, flags: u32) -> i32;
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    }
    const MFD_HUGETLB: u32 = 4;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_SHARED: i32 = 1;
    const MAP_FAILED: *mut u8 = !0usize as *mut u8;
    const HUGE: usize = 2 << 20;

    /// 64-bit hash of the parts, word at a time (about 20 ms for 110 MB).
    fn hash(parts: &[&[u8]]) -> u64 {
        let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
        for p in parts {
            let mut c = p.chunks_exact(8);
            for w in &mut c {
                h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).rotate_left(23).wrapping_mul(0x2127_599b_f432_5c37);
            }
            for &b in c.remainder() {
                h = (h ^ b as u64).wrapping_mul(0x2127_599b_f432_5c37);
            }
            h ^= p.len() as u64;
        }
        h ^ (h >> 29)
    }

    fn map(fd: i32, len: usize, write: bool) -> Option<*mut u8> {
        let prot = if write { PROT_READ | PROT_WRITE } else { PROT_READ };
        let p = unsafe { mmap(std::ptr::null_mut(), len, prot, MAP_SHARED, fd, 0) };
        (p != MAP_FAILED).then_some(p)
    }

    /// Opens a live holder's copy named `name` of `len` bytes, if any.
    fn attach(holders: &str, name: &str, len: usize) -> Option<(File, *const u8)> {
        for l in holders.lines() {
            let path = match l.split_once(' ') {
                Some((pid, fd)) => format!("/proc/{}/fd/{}", pid, fd),
                None => continue,
            };
            // The link names the memfd, so a recycled pid or fd is rejected.
            match std::fs::read_link(&path) {
                Ok(t) if t.to_string_lossy().starts_with(&format!("/memfd:{} ", name)) => {}
                _ => continue,
            }
            let Ok(f) = File::open(&path) else { continue };
            if f.metadata().map(|m| m.len() as usize).ok() != Some(len) {
                continue;
            }
            if let Some(p) = map(f.as_raw_fd(), len, false) {
                return Some((f, p as *const u8));
            }
        }
        None
    }

    /// The shared copy of the parts, laid out back to back with each part
    /// starting on a 2 MB boundary; returns each part's start. None when
    /// disabled or on any failure (the caller keeps its own copy).
    pub fn share(parts: &[&[u8]]) -> Option<Vec<*const u8>> {
        let mode = std::env::var("INCIPIT_NETSHM").unwrap_or_default();
        if mode == "0" {
            return None;
        }
        let offs: Vec<usize> = parts
            .iter()
            .scan(0usize, |o, p| {
                let s = *o;
                *o += p.len().div_ceil(HUGE) * HUGE;
                Some(s)
            })
            .collect();
        let len = offs.last()? + parts.last()?.len().div_ceil(HUGE) * HUGE;
        let name = format!("incipit-net-{:016x}-{}", hash(parts), len);
        let mut reg = OpenOptions::new().read(true).write(true).create(true).open(format!("/dev/shm/{}", name)).ok()?;
        reg.lock().ok()?;
        let mut holders = String::new();
        reg.read_to_string(&mut holders).ok()?;
        let (f, base) = match attach(&holders, &name, len) {
            Some(x) => x,
            None => {
                let cname = format!("{}\0", name);
                let make = |flags| {
                    let fd = unsafe { memfd_create(cname.as_ptr(), flags) };
                    if fd < 0 {
                        return None;
                    }
                    let f = unsafe { File::from_raw_fd(fd) };
                    f.set_len(len as u64).ok()?;
                    // A hugetlb mapping reserves its pages here, or fails.
                    let p = map(f.as_raw_fd(), len, true)?;
                    Some((f, p))
                };
                let (f, p) = (if mode == "4k" { None } else { make(MFD_HUGETLB) }).or_else(|| make(0))?;
                for (part, &o) in parts.iter().zip(&offs) {
                    unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), p.add(o), part.len()) };
                }
                (f, p as *const u8)
            }
        };
        // Every holder keeps its fd open for life and lists itself, so the
        // copy stays findable while any holder runs; dead entries are pruned.
        let fd = f.into_raw_fd();
        let mut list: String = holders
            .lines()
            .filter(|l| {
                l.split_once(' ').is_some_and(|(pid, _)| std::path::Path::new(&format!("/proc/{}", pid)).exists())
            })
            .map(|l| format!("{}\n", l))
            .collect();
        list += &format!("{} {}\n", std::process::id(), fd);
        reg.set_len(0).ok()?;
        reg.seek(SeekFrom::Start(0)).ok()?;
        reg.write_all(list.as_bytes()).ok()?;
        Some(offs.iter().map(|&o| unsafe { base.add(o) as *const u8 }).collect())
    }
}

#[cfg(target_os = "linux")]
pub use imp::share;

#[cfg(not(target_os = "linux"))]
pub fn share(_parts: &[&[u8]]) -> Option<Vec<*const u8>> {
    None
}
