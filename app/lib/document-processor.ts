
import { useAudioStore } from "./audio-store"

// Function to extract text from different file types
export async function processDocument(file: File): Promise<void> {
	try {
		let text = ""

		if (file.type === "application/pdf") {
			text = await extractPdfText(file)
		} else if (file.type === "text/plain") {
			text = await extractTxtText(file)
		} else if (file.type === "application/epub+zip" || file.name.endsWith(".epub")) {
			text = await extractEpubText(file)
		} else {
			throw new Error("Unsupported file type")
		}

		// Clean up the text
		text = cleanText(text)

		// Store the text for audio playback
		useAudioStore.getState().setAudioText(text)

		return
	} catch (error) {
		console.error("Error processing document:", error)
		throw error
	}
}

// Extract text from PDF files
async function extractPdfText(file: File): Promise<string> {
	// In a real implementation, you would use PDF.js
	// This is a simplified version for demonstration

	// Simulate PDF text extraction
	return new Promise((resolve) => {
		setTimeout(() => {
			resolve(`This is simulated text extracted from a PDF file named ${file.name}. 
      In a real implementation, we would use PDF.js to extract the actual text content 
      from the PDF document. The text would include all paragraphs, headings, and other 
      content from the document, properly formatted for text-to-speech conversion.`)
		}, 1500)
	})
}

// Extract text from TXT files
async function extractTxtText(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader()

		reader.onload = (e) => {
			if (e.target?.result) {
				resolve(e.target.result as string)
			} else {
				reject(new Error("Failed to read text file"))
			}
		}

		reader.onerror = () => {
			reject(new Error("Error reading text file"))
		}

		reader.readAsText(file)
	})
}

// Extract text from EPUB files
async function extractEpubText(file: File): Promise<string> {
	// In a real implementation, you would use an EPUB parser
	// This is a simplified version for demonstration

	// Simulate EPUB text extraction
	return new Promise((resolve) => {
		setTimeout(() => {
			resolve(`This is simulated text extracted from an EPUB file named ${file.name}. 
      In a real implementation, we would use an EPUB parser to extract the actual text 
      content from the EPUB document. The text would include all chapters, paragraphs, 
      and other content from the book, properly formatted for text-to-speech conversion.`)
		}, 1500)
	})
}

// Clean up text for better TTS performance
function cleanText(text: string): string {
	return text
		.replace(/\s+/g, " ") // Replace multiple spaces with a single space
		.replace(/\n+/g, " ") // Replace newlines with spaces
		.replace(/\t+/g, " ") // Replace tabs with spaces
		.replace(/\s+\./g, ".") // Fix spacing before periods
		.replace(/\s+,/g, ",") // Fix spacing before commas
		.trim() // Remove leading/trailing whitespace
}
