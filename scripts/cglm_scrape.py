import tqdm, requests
from multiprocessing import Pool

PATH_TO_TRAINATTRIBUTIONS = './train_attribution.csv'

ids = []

fr = open(PATH_TO_TRAINATTRIBUTIONS, 'r')
lines = fr.readlines()
fr.close()

# Get photo_id, line_match fields from the file
for line in tqdm.tqdm(lines):
    photo_id = line.strip().split(',')[1].split(':')[-1].strip()
    line_match = line.strip().split(',')[0].strip()
    ids.append((photo_id, line_match))

# Function to download and store the metadata
def download_metadata(id):
    photo_id, line_match = id
    try:
        res = requests.get('https://opendata.utou.ch/glam/magnus-toolserver/commonsapi.php?meta&image='+photo_id) 
        with open('./metadata/'+line_match+'.txt','w') as f:
            f.write(res.content.decode('utf8')+'\n')
            f.flush()
    except:
        pass

# Set the pool size according to your PC
pool = Pool(64)
for _ in tqdm.tqdm(pool.imap_unordered(download_metadata, ids), total=len(ids)):
    pass