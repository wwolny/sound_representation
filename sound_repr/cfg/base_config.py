from dataclasses import dataclass, fields


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls, obj: dict):
        fields_names = [fld.name for fld in fields(cls)]
        dct = {k: v for (k, v) in obj.items() if k in fields_names}
        return cls(**dct)
