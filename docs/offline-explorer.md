# Reading the offline report

Open `report.html` to see the analyzed observations, references, exclusions, score definitions and input checksums. Follow **Explore results** for the score maps, rank comparisons and searchable tables. Both pages work from a local folder. All plots, styles and controls are packaged locally. The explorer brings in no remote fonts, scripts or images.

```console
triplet report --run analysis --out report-copy --palette accessible --figure-format svg
triplet report --scores analysis/scores --out pair-report --palette amber-blue
triplet report --run analysis --structure mapped-structure --out structure-report --figure-format pdf --figure-format png
```

Use a new output directory. The first command verifies the saved scientific inventory and renders its results. The second uses the full pair-score archive. A pair-only archive supports reranking and displaying the stored pair matrices. Observation audits, triple fits and nested holdouts need their accompanying run artifacts. Older results with only ranked tables can display their available rows. A regenerated workflow report carries the saved matching summary, when available. The original parent workflow directory retains the complete search and matching audits that lived outside the scored run.

## Maps and rank comparisons

Each interface shows the two original, one-based alignment axes. Large maps group adjacent retained sites into at most 28 bins per axis. Labels give source-column endpoints, which can span excluded columns. Each colored bin is the arithmetic mean of its eligible finite cells. Hatching identifies a bin with no eligible finite scores. Colored zero can denote measured zero or locally cancelling signed scores. The captions give the finite cell count and joint-observation support range.

Raw and corrected MI use bits. DCA units come from the archived estimator. A method's scales agree across AB, AC and BC. APC uses a symmetric range about zero; its palette distinguishes both signs. The scale maximum uses the largest absolute eligible individual score across the three blocks, including when an extreme is diluted by averaging. Select a statistic or change the palette in the explorer controls. SVG export requires the base package. PNG and PDF maps require the plots extra. Exports include coordinate labels, units, legends and the mapped-structure threshold when supplied.

The rank comparison intersects the two methods' finite score masks, minimum counts, minimum weights and applicable reference-coordinate restrictions. Each axis assigns competition rank as one plus the number of strictly larger scores **on that same intersection**. Up to 192 coordinate pairs spaced through source-coordinate order are drawn per comparison. The diagonal denotes equal ranks. Overlapping rank points can conceal ties. The complete arrays and underlying support are downloadable.

## Structural and calibration evidence

A mapped structure is bound to the matching reference identities, numbering, strand, sequence and optional original score-manifest digest. A ring means at least one mapped pair in that bin satisfied the recorded inclusive distance threshold. A bin without a ring can include measured distant pairs, unmeasurable geometry or positions outside the mapping. The distance table identifies each author residue, insertion code, chain copy and observability. Use the printed source designation to distinguish experimental, predicted, synthetic or unspecified coordinates. The independent viewer selections use the same selected assembly and model.

The Cytoscape-compatible GraphML records ranked pairs as scored statistical associations. A triple has one factor node and three membership edges. An attached structure adds a separate annotated network that joins physical distances to measured **pairs**, and preserves the original score-only network.

Each label benchmark reports its class coverage, AUROC, average precision, top-K precision and enrichment, measured distances and available uncertainty intervals. Unknown labels retain their own counts. Single-class and insufficient-support results retain their recorded availability. ROC and precision–recall plots show up to 128 threshold points spaced through saved threshold order. Metrics come from the full saved calculation. The benchmark JSON retains every threshold and denominator. Older benchmark files above 16 MiB without a compact display artifact have a full download in place of an inline curve preview.

The support section displays available site entropy and canonical-state coverage, exact duplicate counts, taxonomy/clades, candidate selection and tested versus eligible triples. Local maximum-entropy fits show occupied fractions, prior sensitivity, signed co-information and connected information in separate columns. The local fit tables carry individual failures and convergence residuals. When produced, named null, held-out clade and nested conditional-model reports show their recorded test family, thresholds, uncertainty and fold outcomes. A descriptive ranking or a same-data screen does not establish independent validation.

## Finding and downloading a result

The **Data tables** section links the entire exported TSV or CSV for each preview. The sort statistic, direction and number of exported rows distinguish capped rankings from complete stored matrices. With a saved label snapshot, **Supplied labels with supported scores** joins each submitted coordinate pair to the stored scores, carries its train/test/unspecified split and retains unmapped or unsupported rows with blank scores.

Choose a preview, enter search text, an exact label/status, or a numerical field and inclusive limits, then select **Apply filters**. Previews contain at most 40 rows, 30 fields and 200 characters per cell. For the entire table, download it, then use **Open full local table**. The browser streams the file, checks every row and gives the full count of matches. The visible result is capped at 100 rows, 100 columns and 400 characters per cell. Filters operate on the full records. CSV and TSV quoting, embedded line breaks, CRLF and a UTF-8 BOM are supported. The reader accepts up to 512 fields and 4,194,304 decoded characters per record. The browser never transfers the file. Without JavaScript, charts, static table samples and downloads remain available.

## Rendering scope

The full archive loader verifies hashes and loads numeric matrices in memory. Report aggregation and rank comparisons then scan row tiles, keeping only small grids, a rank sample and short table previews. Map resolution and HTML size stay independent of the number of stored pair cells. Structural distances, QC rows and tabular previews are read without creating a full browser-side table. The generated labeled-score download scales with the number of supplied annotations. The optional Matplotlib exports and initially loading the full score archive add memory beyond the tiled display workspace.

In a synthetic 640-site-per-component check, 4,915,200 stored score cells produced a 270,588-byte HTML explorer. The corrected AB rank comparison included 377,411 jointly supported finite coordinates and drew 192 of them. The fixture used four statistics with 4% missingness per method (NumPy default RNG, seed 24010) and no external evidence. Rendering the in-memory fixture took 0.20 seconds on the test host.
