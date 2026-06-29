import pytest

from nsm import particles


_BASELINE_PARTICLES = dict(particles.PARTICLES)


@pytest.fixture(autouse=True)
def reset_particle_registry():
    """Keep candidate registrations from leaking across tests."""
    particles.PARTICLES.clear()
    particles.PARTICLES.update(_BASELINE_PARTICLES)
    yield
    particles.PARTICLES.clear()
    particles.PARTICLES.update(_BASELINE_PARTICLES)
