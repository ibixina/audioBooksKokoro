
"use client"

import type React from "react"

import { useState } from "react"
import { Upload, FileText, FileType, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { processDocument } from "@/lib/document-processor"
import { useToast } from "@/hooks/use-toast"

export function FileUploader() {
	const [file, setFile] = useState<File | null>(null)
	const [isUploading, setIsUploading] = useState(false)
	const [uploadProgress, setUploadProgress] = useState(0)
	const [error, setError] = useState<string | null>(null)
	const { toast } = useToast()

	const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const selectedFile = e.target.files?.[0] || null

		if (!selectedFile) return

		const fileType = selectedFile.type
		const validTypes = ["application/pdf", "text/plain", "application/epub+zip"]

		if (!validTypes.includes(fileType) && !selectedFile.name.endsWith(".epub")) {
			setError("Please upload a PDF, TXT, or EPUB file")
			setFile(null)
			return
		}

		setError(null)
		setFile(selectedFile)
	}

	const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
		e.preventDefault()

		const droppedFile = e.dataTransfer.files[0]
		if (!droppedFile) return

		const fileType = droppedFile.type
		const validTypes = ["application/pdf", "text/plain", "application/epub+zip"]

		if (!validTypes.includes(fileType) && !droppedFile.name.endsWith(".epub")) {
			setError("Please upload a PDF, TXT, or EPUB file")
			return
		}

		setError(null)
		setFile(droppedFile)
	}

	const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
		e.preventDefault()
	}

	const handleUpload = async () => {
		if (!file) return

		setIsUploading(true)
		setUploadProgress(0)

		// Simulate progress
		const progressInterval = setInterval(() => {
			setUploadProgress((prev) => {
				if (prev >= 90) {
					clearInterval(progressInterval)
					return prev
				}
				return prev + 10
			})
		}, 300)

		try {
			await processDocument(file)
			setUploadProgress(100)

			toast({
				title: "Document processed successfully",
				description: "Your audiobook is ready to play",
			})
		} catch (err) {
			setError("Failed to process document. Please try again.")
			toast({
				variant: "destructive",
				title: "Error",
				description: "Failed to process document",
			})
		} finally {
			clearInterval(progressInterval)
			setIsUploading(false)
		}
	}

	const getFileIcon = () => {
		if (!file) return <Upload className="h-8 w-8 text-muted-foreground" />

		if (file.type === "application/pdf") {
			return <FileText className="h-8 w-8 text-red-500" />
		} else if (file.type === "text/plain") {
			return <FileText className="h-8 w-8 text-blue-500" />
		} else {
			return <FileType className="h-8 w-8 text-green-500" />
		}
	}

	return (
		<Card className="w-full">
			<CardContent className="pt-6">
				<div
					className={`border-2 border-dashed rounded-lg p-8 text-center ${error ? "border-red-400" : "border-muted-foreground/25"
						}`}
					onDrop={handleDrop}
					onDragOver={handleDragOver}
				>
					<div className="flex flex-col items-center justify-center gap-4">
						{getFileIcon()}

						<div className="space-y-2">
							<h3 className="text-lg font-medium">{file ? file.name : "Upload your document"}</h3>
							<p className="text-sm text-muted-foreground">
								{file
									? `${(file.size / (1024 * 1024)).toFixed(2)} MB · ${file.type || "EPUB"}`
									: "Drag and drop or click to upload"}
							</p>
						</div>

						<div className="flex gap-2">
							<Button
								variant="outline"
								onClick={() => document.getElementById("file-upload")?.click()}
								disabled={isUploading}
							>
								Choose File
							</Button>

							{file && (
								<Button onClick={handleUpload} disabled={isUploading}>
									{isUploading ? "Processing..." : "Convert to Audio"}
								</Button>
							)}
						</div>

						<input
							id="file-upload"
							type="file"
							accept=".pdf,.txt,.epub"
							onChange={handleFileChange}
							className="hidden"
						/>
					</div>

					{isUploading && (
						<div className="mt-4 space-y-2">
							<Progress value={uploadProgress} className="h-2 w-full" />
							<p className="text-sm text-muted-foreground">
								{uploadProgress < 100 ? "Processing your document..." : "Processing complete!"}
							</p>
						</div>
					)}
				</div>

				{error && (
					<Alert variant="destructive" className="mt-4">
						<AlertCircle className="h-4 w-4" />
						<AlertDescription>{error}</AlertDescription>
					</Alert>
				)}
			</CardContent>
		</Card>
	)
}
