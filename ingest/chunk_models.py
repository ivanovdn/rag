from pydantic import BaseModel


class PolicyChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_filename: str
    doc_link: str
    section_path: list[str]
    section_display: str
    clause_number: str
    text: str
    char_count: int
    chunk_index: int
    last_updated: str
