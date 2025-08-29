# Coding guide for feature engineering functions

- Avoid using Class unless absolutely needed
- For input, try to use default values to avoid writing if else
- For the naming of input, be specific so when building the pipeline, we know which function should be used on which columns
- When we talk about lag N, we are talking about moving backward from the current row (the last row) to N row above
- Try to create expectations for test cases