from pydantic import BaseModel


class PolicyChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_filename: str
    doc_link: str

    section: str = ""         # "Private Information"
    section_number: str = ""  # "7"
    clause: str = ""          # "Blogging and Social Media"
    clause_number: str = ""   # "7.5"

    section_display: str = "" # "7. Private Information > 7.5. Blogging and Social Media"

    text: str
    char_count: int = 0
    chunk_index: int = 0
    last_updated: str = ""
