# export_list

Writes the full content of a session list to a CSV file in the session
`outputs` folder and attaches it to the chat answer.

## When to call it

| The user asks for... | Argument |
|----------------------|----------|
| the final list, the current list, the filtered list, the candidate list, "download the result" | `which="candidate"` |
| the original list, the initial list, "the list I uploaded", "all the accessions" | `which="original"` |

Do not call it when a list has not been loaded yet; the tool returns an error
asking to load accessions first.

## Result

```json
{
  "status": "ok",
  "message": "Exported the candidate list (87 accessions) to 'candidate_list_20260929_153000.csv'. The file is attached to the answer for download.",
  "list": "candidate",
  "rows": 87,
  "file_name": "candidate_list_20260929_153000.csv",
  "output_files": ["tmp/<session>/outputs/candidate_list_20260929_153000.csv"]
}
```

The agent collects every path listed under `output_files` and attaches those
files to the message shown in the chat. Tell the user the file is attached; do
not paste the CSV content.
