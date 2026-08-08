#!/usr/bin/perl
use strict;

# adapted from Ye, NECTEC
# for Myanmar G2P SMT

my $src = shift;

print "<srcset setid=\"G2P_data\" srclang=\"any\">\n";
print "<doc docid=\"none\" genre=\"8000\" origlang=\"$src\">\n";

# Look for test file in the sibling clean-data directory
open FILE, "../clean-data/test.$src" or die "Cannot open ../clean-data/test.$src: $!";

my $id=1;

while( <FILE> )
{
	chomp;
	
	print "<seg id=\"$id\">$_ </seg>\n";
	$id++;
}

print "</doc>\n</srcset>\n";
close FILE;
