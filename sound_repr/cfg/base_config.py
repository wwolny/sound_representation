from dataclasses import dataclass, fields


@dataclass
class BaseConfig:
    """Dataclass with reading from dict method."""

    @classmethod
    def from_dict(cls, obj: dict):
        """Read values to all fields in dataclass from dict."""
        fields_names = [fld.name for fld in fields(cls)]
        dct = {k: v for (k, v) in obj.items() if k in fields_names}
        return cls(**dct)
