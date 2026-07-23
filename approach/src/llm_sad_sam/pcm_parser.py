"""Parse PCM (Palladio Component Model) architecture models."""

from dataclasses import dataclass
from pathlib import Path
from lxml import etree


@dataclass
class ArchitectureComponent:
    """Represents an architecture component from PCM model."""
    id: str
    name: str
    entity_name: str


def parse_pcm_repository(model_path: str | Path) -> list[ArchitectureComponent]:
    """Parse a PCM .repository file and extract components."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    tree = etree.parse(str(model_path))
    root = tree.getroot()

    components = []
    
    # PCM stores components in components__Repository elements
    # with xsi:type attribute indicating BasicComponent or CompositeComponent
    for elem in root.iter():
        local_name = etree.QName(elem.tag).localname if '}' in elem.tag else elem.tag
        
        # Check for components__Repository elements
        if local_name == "components__Repository":
            # Get xsi:type to check component type
            xsi_type = elem.get("{http://www.w3.org/2001/XMLSchema-instance}type", "")
            if "BasicComponent" in xsi_type or "CompositeComponent" in xsi_type:
                comp_id = elem.get("id", "")
                entity_name = elem.get("entityName", "")
                if comp_id and entity_name:
                    components.append(ArchitectureComponent(
                        id=comp_id,
                        name=entity_name,
                        entity_name=entity_name,
                    ))
    
    return components


def get_component_names(model_path: str | Path) -> dict[str, str]:
    """Get a mapping of component IDs to names."""
    components = parse_pcm_repository(model_path)
    return {comp.id: comp.name for comp in components}


def get_component_name_list(model_path: str | Path) -> list[str]:
    """Get list of component names."""
    components = parse_pcm_repository(model_path)
    return [comp.name for comp in components]
