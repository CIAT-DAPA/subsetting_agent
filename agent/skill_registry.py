"""Automatic discovery and registration of skills.

The registry walks the ``skills`` package, imports every ``skills.<name>.skill``
module and registers each concrete :class:`~skills.base.Skill` subclass found.
Adding a new skill therefore only requires a new folder; no central list has to
be edited.
"""

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any

from core.logger import get_logger
from skills.base import Skill

logger = get_logger(__name__)


class SkillRegistry:
    """Holds the skills available to the agent, indexed by tool name."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._skills: dict[str, Skill] = {}

    # ------------------------------------------------------------- discovery
    def discover(self, package_name: str = "skills") -> "SkillRegistry":
        """Import every skill sub package and register the skills it defines.

        Args:
            package_name: Import path of the package that contains the skills.

        Returns:
            ``self`` to allow chaining (``SkillRegistry().discover()``).
        """
        package: ModuleType = importlib.import_module(package_name)

        # Iterate over the direct children of the skills package. Only sub
        # packages (folders) are considered; ``base.py`` and other modules are skipped.
        for module_info in pkgutil.iter_modules(package.__path__):
            if not module_info.ispkg:
                continue

            module_path = f"{package_name}.{module_info.name}.skill"

            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as exc:
                # A folder without ``skill.py`` is not a skill; warn and move on.
                logger.warning("Skipping %s: %s", module_path, exc)
                continue

            self._register_module(module)

        logger.info("Registered skills: %s", list(self._skills))
        return self

    def _register_module(self, module: ModuleType) -> None:
        """Register every concrete ``Skill`` subclass defined in a module.

        Args:
            module: Imported ``skills.<name>.skill`` module.
        """
        # Inspect every class of the module and keep the concrete skills only.
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            # Typing aliases (e.g. ``Callable[[], X]``) pass ``isclass`` (and even
            # ``isinstance(..., type)`` in Python 3.10) but ``issubclass`` raises on them.
            try:
                is_skill = issubclass(candidate, Skill) and candidate is not Skill
            except TypeError:
                continue

            is_local = getattr(candidate, "__module__", None) == module.__name__

            # Ignore ``Skill`` itself, abstract helpers and re-exported classes.
            if not (is_skill and is_local) or inspect.isabstract(candidate):
                continue

            self.register(candidate())

    def register(self, skill: Skill) -> None:
        """Add a skill instance to the registry.

        Args:
            skill: Skill to register.

        Raises:
            ValueError: If the skill has no name or the name is already taken.
        """
        # Names are the identity of a tool for the LLM; they must be unique.
        if not skill.name:
            raise ValueError(f"Skill {skill.__class__.__name__} has no name.")
        if skill.name in self._skills:
            raise ValueError(f"Duplicated skill name: {skill.name}")

        self._skills[skill.name] = skill
        logger.debug("Registered skill %s", skill.name)

    # --------------------------------------------------------------- queries
    def get(self, name: str) -> Skill | None:
        """Return the skill registered under ``name`` or ``None``."""
        return self._skills.get(name)

    def names(self) -> list[str]:
        """Return the registered tool names in registration order."""
        return list(self._skills)

    def tools(self) -> list[dict[str, Any]]:
        """Return every skill serialised as an OpenAI function tool."""
        return [skill.to_openai_tool() for skill in self._skills.values()]

    def describe(self) -> str:
        """Render a bullet list ``- name: description`` for the system prompt.

        Returns:
            Markdown text; a placeholder sentence when no skill is registered.
        """
        # Without skills the model must know it cannot process data yet.
        if not self._skills:
            return "- (no tools are available yet)"

        return "\n".join(f"- {skill.name}: {skill.description}" for skill in self._skills.values())

    def __len__(self) -> int:
        """Number of registered skills."""
        return len(self._skills)
