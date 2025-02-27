from __future__ import unicode_literals
import json as _json
import os as _os
import sys as _sys
import fire as _fire
import yt_dlp as _yt_dlp

from Solos import SOLOS_IDS_PATH

__all__ = ['YouTubeSaver']


class YouTubeSaver(object):
    def __init__(self):
        self.outtmpl = '%(id)s.%(ext)s'
        self.ydl_opts = {
            'format': 'bestvideo+bestaudio',
            'outtmpl': self.outtmpl,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
                'nopostoverwrites': False, 
            }],
            'keepvideo': True,  
            'logger': None
        }

    def from_json(self, dataset_dir, json_path=SOLOS_IDS_PATH):
        dataset = _json.load(open(json_path))
        dataset_dir = "E:/MUSIC-CMV2M/Solos/data"

        for instrument in dataset.keys():
            if not _os.path.exists(_os.path.join(dataset_dir, instrument)):
                _os.makedirs(_os.path.join(dataset_dir, instrument))
            self.ydl_opts['outtmpl'] = _os.path.join(dataset_dir, instrument, self.outtmpl)
            
            with _yt_dlp.YoutubeDL(self.ydl_opts) as ydl:  # Use yt-dlp's YoutubeDL
                for i, video_id in enumerate(dataset[instrument]):
                    try:
                        ydl.download(['https://www.youtube.com/watch?v=%s' % video_id])
                    except OSError:
                        with open(_os.path.join(dataset_dir, 'backup.json'), 'w') as dst_file:
                            _json.dump(dataset, dst_file)
                        print('Process failed at video {0}, #{1}'.format(video_id, i))
                        print('Backup saved at {0}'.format(_os.path.join(dataset_dir, 'backup.json')))
                    except KeyError:
                        print(f'Skipping missing video ID: {video_id}')
                    except _yt_dlp.utils.DownloadError:
                        print(f'Failed to download video ID: {video_id}')
                    except KeyboardInterrupt:
                        _sys.exit()


if __name__ == '__main__':
    _fire.Fire(YouTubeSaver)

    # USAGE
    # python youtubesaver.py from_json /whereyouwantem
    # THIS NEEDS TO BE IN YOUR SOLOS DIRECTORY
