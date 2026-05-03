from src.ingest import load_data

def test_load():
    df = load_data()
    assert df.shape[1] == 4