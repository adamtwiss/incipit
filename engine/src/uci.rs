//! UCI input helpers: option values as GUIs send them, and path lists.

/// The raw value of a `setoption name <id> value <x>` command: everything
/// after the `value` keyword, spaces kept exactly. A value wrapped in one
/// matched pair of double quotes (as some GUIs send paths) is unwrapped.
/// None when the command has no value part (e.g. a button option).
pub fn option_value(cmd: &str) -> Option<String> {
    // Find the `value` keyword as a whole token, by byte offset, so the rest
    // of the line can be taken verbatim.
    let mut rest = cmd;
    loop {
        let t = rest.trim_start();
        if t.is_empty() {
            return None;
        }
        let end = t.find(char::is_whitespace).unwrap_or(t.len());
        let (tok, after) = t.split_at(end);
        if tok.eq_ignore_ascii_case("value") {
            let v = after.trim();
            let v = match v.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
                Some(inner) => inner,
                None => v,
            };
            return Some(v.to_string());
        }
        rest = after;
    }
}

/// Splits a list of directories on the platform's PATH-list separator (':' on
/// Unix, ';' on Windows, where ':' also appears in drive letters). Empty
/// segments are dropped; spaces belong to the names.
pub fn split_paths(s: &str) -> Vec<String> {
    split_paths_with(s, if cfg!(windows) { ';' } else { ':' })
}

fn split_paths_with(s: &str, sep: char) -> Vec<String> {
    s.split(sep).filter(|p| !p.is_empty()).map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_with_spaces_is_kept_whole() {
        assert_eq!(
            option_value(r"setoption name SyzygyPath value C:\Program Files\Syzygy").as_deref(),
            Some(r"C:\Program Files\Syzygy")
        );
    }

    #[test]
    fn value_in_matched_double_quotes_is_unwrapped() {
        assert_eq!(
            option_value(r#"setoption name SyzygyPath value "C:\Program Files\Syzygy""#).as_deref(),
            Some(r"C:\Program Files\Syzygy")
        );
    }

    #[test]
    fn value_with_a_lone_quote_is_unchanged() {
        assert_eq!(option_value(r#"setoption name Book value "book.bin"#).as_deref(), Some(r#""book.bin"#));
    }

    #[test]
    fn command_without_value_has_none() {
        assert_eq!(option_value("setoption name Ponder"), None);
    }

    #[test]
    fn value_keyword_is_case_insensitive_and_name_may_contain_value_like_text() {
        assert_eq!(option_value("setoption name Hash VALUE 64").as_deref(), Some("64"));
        assert_eq!(option_value("setoption name ValueWeight value 3").as_deref(), Some("3"));
    }

    #[test]
    fn one_directory() {
        assert_eq!(split_paths_with("/tablebases", ':'), vec!["/tablebases"]);
    }

    #[test]
    fn empty_list_has_no_directories() {
        assert!(split_paths_with("", ':').is_empty());
        assert!(split_paths("").is_empty());
    }

    #[test]
    fn colon_separated_directories() {
        assert_eq!(split_paths_with("/tb/3-4-5:/tb/6:/tb/7", ':'), vec!["/tb/3-4-5", "/tb/6", "/tb/7"]);
    }

    #[test]
    fn spaces_are_part_of_directory_names() {
        assert_eq!(split_paths_with("/my tables/6 man:/other tb", ':'), vec!["/my tables/6 man", "/other tb"]);
    }

    #[test]
    fn empty_segments_are_skipped() {
        assert_eq!(split_paths_with("/tb/a::/tb/b", ':'), vec!["/tb/a", "/tb/b"]);
    }

    #[test]
    fn semicolon_list_keeps_drive_letter_colons() {
        assert_eq!(split_paths_with(r"C:\tb\6;D:\tb\7", ';'), vec![r"C:\tb\6", r"D:\tb\7"]);
    }

    #[cfg(unix)]
    #[test]
    fn platform_separator_is_colon_on_unix() {
        assert_eq!(split_paths("/tb/3-4-5:/tb/6"), vec!["/tb/3-4-5", "/tb/6"]);
    }

    #[cfg(windows)]
    #[test]
    fn platform_separator_is_semicolon_on_windows() {
        assert_eq!(split_paths(r"C:\tb\6;D:\tb\7"), vec![r"C:\tb\6", r"D:\tb\7"]);
    }
}
