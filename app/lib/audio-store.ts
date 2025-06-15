
import { create } from "zustand"

interface AudioState {
	audioText: string | null
	isReady: boolean
	setAudioText: (text: string) => void
	reset: () => void
}

export const useAudioStore = create<AudioState>((set) => ({
	audioText: null,
	isReady: false,
	setAudioText: (text) => set({ audioText: text, isReady: true }),
	reset: () => set({ audioText: null, isReady: false }),
}))
