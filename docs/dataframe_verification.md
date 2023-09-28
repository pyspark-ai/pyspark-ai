# DataFrame Verification

## API
```python
DataFrame.ai.verify(desc: Optional[str] = None, cache: bool = True) -> None
```
This method creates and runs test cases for the provided PySpark dataframe transformation function. The result is shown in the logging output.

## Example
Given a DataFrame `auto_df` from [Data Ingestion](data_ingestion.md):
```python
auto_df.ai.verify("expect sales change percentage to be between -100 to 100")
```

> result: True
