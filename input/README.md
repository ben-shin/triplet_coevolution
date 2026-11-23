# Three-family input

Create three folders here and name each after its protein or nucleic acid. Put one Stockholm alignment in each folder. The files you add here are ignored by Git. The pipeline reads them in place when you invoke it. An example layout is shown below.

```text
input/
  ATPase/family.sto.gz
  Partner/family.sto
  guide_RNA/family.sto.gz
```

Run the following command from the repository root to collect offline evidence and prepare selected rows. Results go to a new directory.

```shell
triplet evidence --input input --out results/prepared
```

You can start with the alignments alone to inventory their rows. To obtain resolved groups, supply identity and genome or assembly evidence for the candidates, and identify each component's role using `expected_orthogroups` in `input/project.json` or a `reference_id` whose orthogroup the resolver can establish. A group needs one unambiguous, role-supported member from each family in the same genomic context. The [example project](../examples/families/project.json) shows the three role assignments.

For an offline project, supply a root `mappings.tsv` following the [mapping example](../examples/families/mappings.tsv), or point the project to a previously saved UniProt cache for its protein components. Include curated locus mappings for DNA and RNA. Inspect `metadata/groups.tsv` for the selected groups and `metadata/evidence.tsv` for missing or conflicting evidence. A preparation with zero resolved groups still writes its audit.

For three protein families, you can enable live UniProt lookups explicitly. For mixed families, declare the molecule in each folder's `component.json`, then use `--uniprot` without the default type. A `targets.txt` in any folder limits resolution to the identifiers you list there.

```shell
triplet evidence --input input --default-molecule protein --uniprot --out results/protein-evidence
```

To prepare for scoring, put a file such as `{"molecule":"rna","reference_id":"my_reference"}` in the relevant folder as `component.json`. Supply a reference and molecule for each component. The [workflow guide](../docs/families.md) explains ambiguous domain copies, expected orthogroups, caches, and how to assign folder names explicitly to A, B and C. The [small offline example](../examples/families/) includes a complete mixed-molecule project and synthetic mapping records.
