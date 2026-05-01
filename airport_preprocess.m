A = readtable('dataset/airports/T_T100D_SEGMENT_ALL_CARRIER.csv');
d = configureDictionary("double", "double");

mask = (A.MONTH <= 3);

orig = A.ORIGIN_AIRPORT_SEQ_ID(mask);
dest = A.DEST_AIRPORT_SEQ_ID(mask);

airport_count = numel(unique([orig; dest]));
disp(airport_count)