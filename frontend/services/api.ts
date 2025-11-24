import { RetrievalSteps } from '../types';

const API_URL = 'http://localhost:8000/api/chat';

export const sendMessageToBackend = async (text: string): Promise<{ answer: string; steps: RetrievalSteps }> => {
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

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
};
