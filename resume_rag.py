"""Part A: index resumes into a Chroma vector store.

Planned subcommands (see DESIGN.md and project-plan.md, step 8):

- ``index`` - load resumes from ``WORKSPACE_ROOT`` via ``fs_tools``, detect sections, chunk,
  extract metadata, embed and upsert into Chroma (unchanged files are skipped).
- ``stats`` - report the indexed candidates, chunk counts and collection configuration.
"""
