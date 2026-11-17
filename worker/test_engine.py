from football_engine import build_sequence_features,event_candidates

def test_empty():
 r=build_sequence_features([]);assert r['frames']==0 and r['windows']==[]

def test_candidate():
 r=build_sequence_features([{'timestamp':0},{'timestamp':.5}]);assert r['frames']==2;assert isinstance(event_candidates(r),list)