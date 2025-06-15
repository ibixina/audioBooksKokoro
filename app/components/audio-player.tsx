
"use client"

import { useState, useEffect } from "react"
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { useAudioStore } from "@/lib/audio-store"

export function AudioPlayer() {
	const { audioText, isReady } = useAudioStore()
	const [isPlaying, setIsPlaying] = useState(false)
	const [currentTime, setCurrentTime] = useState(0)
	const [duration, setDuration] = useState(0)
	const [isMuted, setIsMuted] = useState(false)
	const [playbackRate, setPlaybackRate] = useState(1)
	const [utterance, setUtterance] = useState<SpeechSynthesisUtterance | null>(null)

	// Track progress
	useEffect(() => {
		if (!isReady || !audioText) return

		const newUtterance = new SpeechSynthesisUtterance(audioText)

		newUtterance.onstart = () => {
			setIsPlaying(true)
		}

		newUtterance.onend = () => {
			setIsPlaying(false)
			setCurrentTime(0)
		}

		newUtterance.onpause = () => {
			setIsPlaying(false)
		}

		newUtterance.onresume = () => {
			setIsPlaying(true)
		}

		// Estimate duration (rough approximation)
		// Average reading speed is about 150 words per minute
		const wordCount = audioText.split(/\s+/).length
		const estimatedDuration = (wordCount / 150) * 60
		setDuration(estimatedDuration)

		setUtterance(newUtterance)

		return () => {
			window.speechSynthesis.cancel()
		}
	}, [audioText, isReady])

	// Update progress bar
	useEffect(() => {
		if (!isPlaying || !utterance) return

		const interval = setInterval(() => {
			setCurrentTime((prev) => {
				if (prev >= duration) {
					clearInterval(interval)
					return 0
				}
				return prev + 0.1
			})
		}, 100)

		return () => clearInterval(interval)
	}, [isPlaying, duration, utterance])

	// Handle playback rate changes
	useEffect(() => {
		if (utterance) {
			utterance.rate = playbackRate
		}
	}, [playbackRate, utterance])

	const togglePlayPause = () => {
		if (!utterance || !audioText) return

		if (isPlaying) {
			window.speechSynthesis.pause()
			setIsPlaying(false)
		} else {
			if (currentTime === 0 || window.speechSynthesis.paused) {
				if (window.speechSynthesis.paused) {
					window.speechSynthesis.resume()
				} else {
					window.speechSynthesis.speak(utterance)
				}
				setIsPlaying(true)
			}
		}
	}

	const handleSeek = (value: number[]) => {
		// Note: Web Speech API doesn't support precise seeking
		// This is a simplified implementation
		setCurrentTime(value[0])

		if (isPlaying) {
			window.speechSynthesis.cancel()

			if (utterance && audioText) {
				// Create a new utterance with the text starting from an approximated position
				const words = audioText.split(/\s+/)
				const wordsPerSecond = words.length / duration
				const startWordIndex = Math.floor(value[0] * wordsPerSecond)
				const remainingText = words.slice(startWordIndex).join(" ")

				const newUtterance = new SpeechSynthesisUtterance(remainingText)
				newUtterance.rate = playbackRate
				setUtterance(newUtterance)

				window.speechSynthesis.speak(newUtterance)
			}
		}
	}

	const toggleMute = () => {
		if (!utterance) return

		utterance.volume = isMuted ? 1 : 0
		setIsMuted(!isMuted)
	}

	const formatTime = (seconds: number) => {
		const mins = Math.floor(seconds / 60)
		const secs = Math.floor(seconds % 60)
		return `${mins}:${secs.toString().padStart(2, "0")}`
	}

	const skipForward = () => {
		const newTime = Math.min(currentTime + 10, duration)
		handleSeek([newTime])
	}

	const skipBackward = () => {
		const newTime = Math.max(currentTime - 10, 0)
		handleSeek([newTime])
	}

	const changePlaybackRate = () => {
		const rates = [0.75, 1, 1.25, 1.5, 1.75, 2]
		const currentIndex = rates.indexOf(playbackRate)
		const nextIndex = (currentIndex + 1) % rates.length
		setPlaybackRate(rates[nextIndex])
	}

	if (!isReady) {
		return (
			<Card>
				<CardContent className="p-6">
					<div className="flex flex-col items-center justify-center py-8 text-center">
						<h3 className="text-lg font-medium mb-2">No audio available</h3>
						<p className="text-sm text-muted-foreground">Upload a document to generate an audiobook</p>
					</div>
				</CardContent>
			</Card>
		)
	}

	return (
		<Card>
			<CardContent className="p-6">
				<div className="space-y-4">
					<div className="flex items-center justify-between">
						<div className="text-sm text-muted-foreground">{formatTime(currentTime)}</div>
						<div className="text-sm text-muted-foreground">{formatTime(duration)}</div>
					</div>

					<Slider value={[currentTime]} max={duration} step={0.1} onValueChange={handleSeek} className="w-full" />

					<div className="flex items-center justify-between">
						<Button variant="outline" size="icon" onClick={toggleMute} className="rounded-full">
							{isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
						</Button>

						<div className="flex items-center gap-2">
							<Button variant="outline" size="icon" onClick={skipBackward} className="rounded-full">
								<SkipBack className="h-4 w-4" />
							</Button>

							<Button size="icon" onClick={togglePlayPause} className="h-12 w-12 rounded-full">
								{isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6 ml-1" />}
							</Button>

							<Button variant="outline" size="icon" onClick={skipForward} className="rounded-full">
								<SkipForward className="h-4 w-4" />
							</Button>
						</div>

						<Button variant="outline" size="sm" onClick={changePlaybackRate} className="text-xs">
							{playbackRate}x
						</Button>
					</div>
				</div>
			</CardContent>
		</Card>
	)
}

