from dataclasses import dataclass, field


@dataclass()
class DownloadParams:
    gdrive_id: str
    output_folder: str
    zip_name: str = field(default="downloaded.zip")
