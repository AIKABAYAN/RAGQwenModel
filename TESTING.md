# Testing

This directory contains tests for the RAGQwenModel project.

## Test Files

- `test_database.py` - Unit tests for database configuration and operations
- `test_database_integration.py` - Integration tests that connect to a real database
- `run_tests.py` - Test runner script

## Running Tests

### Run all unit tests (excluding integration tests):
```bash
python run_tests.py
```

### Run integration tests (requires database connection):
```bash
python run_tests.py --integration
```

### Run all tests:
```bash
python run_tests.py --all
```

## Test Configuration

The tests use the same database configuration as the main application, which is loaded from the `.env` file:
- Database name: `infinity_cafe`
- User: `admin`
- Password: `password`
- Host: `localhost`
- Port: `5432`

Integration tests will be skipped if the database is not available or accessible.