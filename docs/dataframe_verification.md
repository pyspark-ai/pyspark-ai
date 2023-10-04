# DataFrame Verification

## API
```python
DataFrame.ai.verify(desc: Optional[str] = None, cache: bool = True) -> bool
```
This method creates and runs test cases for the provided PySpark dataframe transformation function. The result is shown in the logging output. The mthod returns `True` if tranformation is valid, otherwise it returns `False` and logs errors.

## Example
Given a DataFrame `auto_df` from [Data Ingestion](data_ingestion.md):
```python
auto_df.ai.verify("expect sales change percentage to be between -100 to 100")
```

> result: True
