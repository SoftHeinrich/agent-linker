from dataclasses import dataclass
from xml.etree import ElementTree

XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"


@dataclass(frozen=True)
class Sentence:
    number: int
    text: str


@dataclass(frozen=True)
class Component:
    id: str
    name: str


@dataclass(frozen=True)
class Link:
    sentence: int
    component_id: str
    component_name: str
    source: str


def load_sentences(path):
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    return [Sentence(number, text)
            for number, text in enumerate((line for line in lines if line), 1)]


def load_components(path):
    components = []
    for element in ElementTree.parse(path).getroot().iter():
        if element.tag.rsplit("}", 1)[-1] != "components__Repository":
            continue
        kind = element.get(XSI_TYPE, "")
        if "BasicComponent" not in kind and "CompositeComponent" not in kind:
            continue
        if element.get("id") and element.get("entityName"):
            components.append(Component(element.get("id"), element.get("entityName")))
    return components
