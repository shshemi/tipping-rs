use itertools::Itertools;
use regex::Regex;


pub fn compile_into_regex<Item, Iter>(regex_str: Iter) -> Regex
where
 Item: AsRef<str>,
 Iter: Iterator<Item = Item>
{
    Regex::new(
        regex_str
            .map(|s| format!(r"(?:{})", s.as_ref()))
            .join("|")
            .as_str(),
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        let r = compile_into_regex([r"\d+", r"[a-zA-Z]+"].into_iter());
        assert!(r.is_match("123"));
        assert!(r.is_match("abc"));
        assert!(r.is_match("ABC"));

        assert!(!r.is_match("@"));
        assert!(!r.is_match("#"));
    }
}