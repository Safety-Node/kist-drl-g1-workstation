"""
KIST DRL G1 demo scenarios — Python-native scenario scripts.

Each module under this package defines one ``Scenario`` (see
:mod:`providers.task_srv_provider`). Add a new scenario by:

1. Creating ``my_scenario.py`` in this directory.
2. Defining a module-level ``Scenario`` object.
3. Appending it to ``ALL`` below.

``TaskSrvProvider.start()`` imports ``ALL`` lazily so adding/removing
scenarios does not change the provider code. Triggers must be unique
across the whole list (TaskSrvProvider raises ``ValueError`` on collision).
"""

from .refrigerator_pickup import refrigerator_pickup

ALL = [
    refrigerator_pickup,
]
