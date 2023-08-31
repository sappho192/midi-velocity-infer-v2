# midi-csv

https://github.com/sappho192/midi-csv
Utility program which converts data between MIDI and CSV files.

## Requirements

- For development: .NET SDK 6.0
- For end-user: .NET 6.0 Runtime

Download link: https://dotnet.microsoft.com/ko-kr/download/dotnet/6.0

Of course there are many other types of data you can retrieve from MIDI, but since this program is made for my another personal project, only mentioned data below are manipulated.
But you can always modify the source code to make your own program from this.

## midi2csv

Reads MIDI file and saves following data into CSV file:

- Timestamp (time)
- Timestamp difference with previous note (time_diff)
- Note Number (note_num)
- Note Number difference (note_num_diff)
- Low Octave (low_locave)
- Length (length)
- Velocity (velocity)

Note that Length is calculated value by subtracting NoteOff and NoteOn time.
Every data type is integer, but range is different:

- time, time_diff, length: Maybe int32?
- note_num, velocity: [0,127]
- note_num_diff: [-127,127]
- low_octave: [0,1]

The value of low_octave will be 1 if the value of note_num is lesser than 72.

Example of csv text file:

```csv
time,time_diff,note_num,note_num_diff,low_octave,length,velocity
755,0,69,0,1,1260,45
784,29,57,-12,1,454,29
```

## csv2midi

Reads CSV file and generates MIDI file.
CSV file must contain following column data:

- Velocity (velocity)
