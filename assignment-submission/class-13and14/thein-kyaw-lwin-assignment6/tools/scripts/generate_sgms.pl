#!/usr/bin/perl
use strict;

# adapted from Ye, NECTEC
# for Myanmar G2P SMT

my @langs = ('my', 'ph');

foreach my $lang (@langs)
{ 
    print "Generating SGM files for language: $lang\n";
    # Output the SGM files directly into the sibling clean-data directory
    `perl ./ref2sgm.pl $lang > ../clean-data/test.$lang.ref.sgm`;
    `perl ./src2sgm.pl $lang > ../clean-data/test.$lang.src.sgm`;
}
print "SGM generation complete!\n";
