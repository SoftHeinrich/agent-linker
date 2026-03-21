"""PCM parser for the S-Linker family (v2).

Clean-slate replacement for pcm_parser.py — lean ArchitectureComponent
(no redundant entity_name), no dead convenience functions.
"""

from dataclasses import dataclass
from pathlib import Path
from lxml import etree


@dataclass
class ArchitectureComponent:
    """An architecture component from a PCM model."""
    id: str
    name: str


def parse_pcm_repository(model_path: str | Path) -> list[ArchitectureComponent]:
    """Parse a PCM .repository file and extract components."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    tree = etree.parse(str(model_path))
    root = tree.getroot()

    components = []

    for elem in root.iter():
        local_name = etree.QName(elem.tag).localname if '}' in elem.tag else elem.tag

        if local_name == "components__Repository":
            xsi_type = elem.get("{http://www.w3.org/2001/XMLSchema-instance}type", "")
            if "BasicComponent" in xsi_type or "CompositeComponent" in xsi_type:
                comp_id = elem.get("id", "")
                entity_name = elem.get("entityName", "")
                if comp_id and entity_name:
                    components.append(ArchitectureComponent(
                        id=comp_id,
                        name=entity_name,
                    ))

    return components
