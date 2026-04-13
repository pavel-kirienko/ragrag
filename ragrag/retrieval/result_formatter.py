"""Format search results as JSON or Markdown.

Provides serialization and human-readable formatting for SearchResponse objects.
"""

from ragrag.models import SearchResponse


def format_as_json(response: SearchResponse) -> str:
    """Serialize SearchResponse to JSON string.
    
    Args:
        response: SearchResponse object to serialize
        
    Returns:
        JSON string with 2-space indentation
    """
    return response.model_dump_json(indent=2)


def format_as_markdown(response: SearchResponse) -> str:
    """Format SearchResponse as human-readable Markdown.
    
    Args:
        response: SearchResponse object to format
        
    Returns:
        Markdown-formatted string with results, indexing stats, and timing info
    """
    lines = []
    
    # Header with query
    lines.append(f"# Search Results: \"{response.query}\"")
    lines.append("")
    
    # Status line
    status_line = f"**Status**: {response.status} | **Total**: {len(response.results)} results | **Time**: {response.timing_ms.total_ms}ms"
    lines.append(status_line)
    lines.append("")
    
    # Indexing stats section
    lines.append("## Indexing")
    stats = response.indexed_now
    lines.append(f"- Added: {stats.files_added} files | Updated: {stats.files_updated} | Skipped (unchanged): {stats.files_skipped_unchanged}")
    lines.append("")
    
    # Results section
    if response.results:
        lines.append("## Results")
        lines.append("")
        
        for result in response.results:
            # Result header with rank, path, and score
            lines.append(f"### {result.rank}. {result.path} (score: {result.score})")
            
            # Metadata line
            metadata_parts = [f"**Type**: {result.file_type}", f"**Modality**: {result.modality}"]
            
            # Add location info based on file type
            if result.page is not None:
                metadata_parts.append(f"**Page**: {result.page}")
            elif result.start_line is not None and result.end_line is not None:
                metadata_parts.append(f"**Lines**: {result.start_line}-{result.end_line}")
            
            lines.append(" | ".join(metadata_parts))
            
            # Excerpt (truncated to 200 chars)
            excerpt = result.excerpt
            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "..."
            lines.append(f"> {excerpt}")
            lines.append("")
    
    # Skipped files section
    if response.skipped_files:
        lines.append("## Skipped Files")
        lines.append("")
        for skipped in response.skipped_files:
            lines.append(f"- {skipped.path}: {skipped.reason}")
        lines.append("")
    
    # Errors section
    if response.errors:
        lines.append("## Errors")
        lines.append("")
        for error in response.errors:
            lines.append(f"- {error}")
        lines.append("")
    
    return "\n".join(lines)
