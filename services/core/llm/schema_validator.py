from jsonschema import validate, ValidationError

def validate_or_fail(data, schema, label="schema"):
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        raise RuntimeError(f"❌ {label} validation failed: {e.message}")
