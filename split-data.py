from pydub import AudioSegment
from pydub.utils import make_chunks

import os

def split_file(name):
    myaudio = AudioSegment.from_file(name , "wav") 
    chunk_length_ms = .3 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
#        altered_chunk = chunk._spawn(chunk.raw_data, overrides={
#           "frame_rate": int(chunk.frame_rate * 4)
#        })
#        altered_chunk.set_frame_rate(chunk.frame_rate)
        chunk_name = name+"chunk{0}.wav".format(i)
        print("exporting" +  str(chunk_name))
        chunk.export(chunk_name, format="wav")

for track in os.listdir("data/wav"):
    if track.endswith("wav"):
        split_file("data/wav/"+track)

os.system("mkdir data/wav/chunks")
os.system("mv data/wav/*chunk*wav data/wav/chunks")
for track in os.listdir("data/wav/chunks"):
  path = "data/wav/chunks/" + track
  if os.path.getsize(path) < 8192*4:
      print("Deleting bad sized chunk" + str(os.path.getsize(path)))
      os.system("rm " + path)
