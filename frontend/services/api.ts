import { RetrievalSteps } from '../types';

const API_URL = 'http://localhost:8000/api/chat';

export const sendMessageToBackend = async (
    text: string,
    onUpdate: (steps: RetrievalSteps) => void
): Promise<void> => {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: text }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        if (!response.body) {
            throw new Error('Response body is null');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');

            // Process all complete messages
            buffer = lines.pop() || ''; // Keep the last incomplete chunk in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.substring(6);
                    try {
                        const data = JSON.parse(jsonStr);
                        if (data.error) {
                            console.error("Backend error:", data.error);
                            // Handle error if needed
                        } else {
                            onUpdate(data);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
};
