Three component coevolution analysis software. This will do inference (MI, APC, and DCA) as pairs and triplets. Accepts protein, DNA, and RNA sequences.
# Workflow
1. 3 component definition - molecule type, search program, database
2. Search, fetch, align
3. Cognate match - homolog identification
4. MSA
5. Preprocessing - filter bad alignments, sequence weighting, evolutionary depth estimation
6. Inference - MI, DCA, three body Potts
7. Signal benchmarking
8. Output stuff (stats, plots, etc.)

# Modes
1. `analyze` - input three matches MSAs and mapping metadata; statistical minimum viable product
2. `prepare` - input three unpaired MSAs + genome/orthology evidence; homolog pairing and QC
3. `run` - input profiles, search databases, reference sequences; complete pipeline (automate eventually)
