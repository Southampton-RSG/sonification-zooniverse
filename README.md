# sonification-zooniverse
Labelling and classifying astro data using the Zooniverse.

Currently, this just generates a base sine wave with magnitude 1, then low, medium and high-noise versions.
As wav files clamp at 1, we need to rescale the amplitude of the wave as we up the noise - or just start with quieter waves.
