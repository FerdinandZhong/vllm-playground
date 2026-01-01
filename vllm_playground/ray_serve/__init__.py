"""
Ray Serve Backend for vLLM Playground
Provides Ray-based orchestration for vLLM deployments
"""

from .ray_backend import RayBackend, ray_backend

__all__ = ['RayBackend', 'ray_backend']
