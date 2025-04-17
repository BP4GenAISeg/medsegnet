from typing import Any, Iterable, Optional, Type
import os
from sympy import Union


def ensure(condition: bool, exc: Type[Exception], msg: str) -> None:
    """General-purpose validation check"""
    if not condition:
        raise exc(msg)

def ensure_has_attr(obj, attr: str, exc: Type[Exception], msg: Optional[str] = None) -> None:
    """Check if our config has attribute, raising specified exception if not"""
    if not hasattr(obj, attr):
        raise exc(msg or f"Required attribute '{attr}' missing from {type(obj).__name__}")
    
def ensure_has_attrs(obj, attrs: list[str], exc: Type[Exception], msg: Optional[str] = None) -> None:
    """Check if our config has attributes, raising specified exception if not"""
    for attr in attrs:
        ensure_has_attr(obj, attr, exc, msg)

def ensure_is_instance(obj, cls: Type, exc: Type[Exception], msg: Optional[str] = None) -> None:
    """Check if our config is an instance of a class, raising specified exception if not"""
    if not isinstance(obj, cls):
        raise exc(msg or f"Expected {cls.__name__}, got {type(obj).__name__}")
        
def ensure_in(obj: Any, collection: Iterable, exc: Type[Exception], msg: Optional[str] = None) -> None:
    """Check if our config is in a collection, raising specified exception if not"""
    if obj not in collection:
        raise exc(msg or f"Expected one of {collection}, got {obj}")

def ensure_pexists(path: str, exc: Type[Exception], msg: Optional[str] = None) -> None:
    """Check if a path exists, raising specified exception if not"""
    if not os.path.exists(path):
        raise exc(msg or f"Path does not exist: {path}")
    
    