from football_engine import build_sequence_features,event_candidates

def test_empty():
 r=build_sequence_features([]);assert r['frames']==0 and r['windows']==[]

def test_candidate():
 r=build_sequence_features([{'timestamp':0},{'timestamp':.5}]);assert r['frames']==2;assert isinstance(event_candidates(r),list)

def test_structural_context_is_camera_space():
    from football_engine import _structural_context
    detections = [
        {
            'timestamp': 10.0,
            'ball': {'x': 500, 'y': 500},
            'detections': [
                {'id': 1, 'class_id': 1, 'team': 'team_a', 'x': 450, 'y': 500},
                {'id': 2, 'class_id': 1, 'team': 'team_a', 'x': 620, 'y': 500},
                {'id': 3, 'class_id': 1, 'team': 'team_a', 'x': 300, 'y': 300},
                {'id': 4, 'class_id': 1, 'team': 'team_b', 'x': 520, 'y': 500},
            ],
        }
    ]
    ev = {'timestamp': 10.0, 'from_team': 'team_a', 'to_team': 'team_b'}
    r = _structural_context(ev, detections, 1000, 800)
    assert r['camera_space'] is True
    assert r['nearest_support_distance_px'] is not None
    assert r['evidence_level'] == 'structural_geometry'
