#!/usr/bin/perl
use strict;

# adapted from Ye, NECTEC
# for Myanmar G2P SMT

my $trg = shift;

print "<refset trglang=\"$trg\" setid=\"G2P_data\" srclang=\"any\">\n";
print "<doc sysid=\"ref\" docid=\"none\" genre=\"8000\" origlang=\"any\">\n";

# Look for test file in the sibling clean-data directory
open FILE, "../clean-data/test.$trg" or die "Cannot open ../clean-data/test.$trg: $!";
             
my $id=1;

while( <FILE> )
{
	chomp;
	
	print "<seg id=\"$id\">$_ </seg>\n";
	$id++;
}

print "</doc>\n</refset>\n";
close FILE;
