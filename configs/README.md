# Configuration policy

Configurations are split by responsibility so a model change is not confused
with a selector or data-partition change.

- data contains dataset paths, labels, preprocessing, and client manifests;
- model contains architecture and loss-independent model settings;
- federation contains communication rounds, local training, and aggregation;
- selection contains participation policy settings;
- experiments pins one complete study by referencing the other layers.

Dataset locations in committed files must be symbolic or relative. Personal
HPC paths belong in environment variables or ignored local override files.

Every reported run must save the fully resolved configuration in its output
directory. Editing a configuration after submission must not alter an existing
run's recorded configuration.
