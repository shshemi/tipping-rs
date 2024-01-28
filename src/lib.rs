mod graph;
mod interdependency;
mod template;
mod tokenizer;
mod traits;
mod parser;
mod token_filter;
pub use parser::Parser;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
