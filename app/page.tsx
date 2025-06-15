
import { FileUploader } from "@/components/file-uploader"
import { AudioPlayer } from "@/components/audio-player"

export default function Home() {
	return (
		<main className="container mx-auto px-4 py-8 max-w-4xl">
			<h1 className="text-3xl font-bold text-center mb-2">Document to Audiobook Converter</h1>
			<p className="text-center text-muted-foreground mb-8">
				Upload your PDF, TXT, or EPUB file and listen to it as an audiobook
			</p>

			<div className="grid gap-8">
				<FileUploader />
				<AudioPlayer />
			</div>
		</main>
	)
}
