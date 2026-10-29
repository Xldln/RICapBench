"""Independent spatial relation extraction for English captions."""
from .parser import PARSER_VERSION, SPATIAL_PREDICATES, parse_captions

__all__ = ["PARSER_VERSION", "SPATIAL_PREDICATES", "parse_captions"]
