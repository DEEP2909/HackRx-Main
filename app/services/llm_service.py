import openai
from typing import List, Dict, Any
from loguru import logger
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential
import json

from app.core.config import settings

class LLMService:
    """Enhanced LLM service optimized for high accuracy insurance document Q&A"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = "gpt-4.1"  # Use GPT-4o-mini for better accuracy
        self.max_tokens = 2000  # Increased for more detailed answers
        self.temperature = 0.0  # Deterministic responses for accuracy
        openai.api_key = self.api_key
        
        # Token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Token limits for optimization
        self.max_context_tokens = 6000  # Increased context window
        self.max_prompt_tokens = 8000   # Increased prompt limit
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate_context(self, context: List[Dict[str, Any]], max_tokens: int = 6000) -> List[Dict[str, Any]]:
        """Intelligently truncate context to stay within token limits"""
        total_tokens = 0
        truncated_context = []
        
        # Sort by relevance score (highest first)
        sorted_context = sorted(context, key=lambda x: x.get('score', 0), reverse=True)
        
        for item in sorted_context:
            content = item['content']
            content_tokens = self.count_tokens(content)
            
            if total_tokens + content_tokens <= max_tokens:
                truncated_context.append(item)
                total_tokens += content_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - total_tokens - 100  # Leave buffer
                if remaining_tokens > 300:  # Only if meaningful space left
                    # Truncate content to fit
                    words = content.split()
                    truncated_words = []
                    temp_tokens = 0
                    
                    for word in words:
                        word_tokens = self.count_tokens(word + " ")
                        if temp_tokens + word_tokens <= remaining_tokens:
                            truncated_words.append(word)
                            temp_tokens += word_tokens
                        else:
                            break
                    
                    if len(truncated_words) > 20:  # Only if we got meaningful content
                        truncated_item = item.copy()
                        truncated_item['content'] = ' '.join(truncated_words) + "..."
                        truncated_context.append(truncated_item)
                
                break
        
        logger.info(f"Context optimized: {len(context)} -> {len(truncated_context)} items, ~{total_tokens} tokens")
        return truncated_context
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_answer(self, question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-accuracy answer using enhanced prompting techniques"""
        
        # Step 1: Optimize context
        optimized_context = self.truncate_context(context, self.max_context_tokens)
        
        if not optimized_context:
            return {
                'answer': 'No relevant information found in the document to answer this question.',
                'confidence': 0.1,
                'relevant_clauses': [],
                'explanation': 'No context available',
                'found_in_context': False,
                'token_usage': {'total_tokens': 0}
            }
        
        # Step 2: Create enhanced prompt for accuracy
        prompt = self._create_enhanced_prompt(question, optimized_context)
        
        # Step 3: Check token count
        prompt_tokens = self.count_tokens(prompt)
        logger.info(f"Enhanced prompt tokens: {prompt_tokens}")
        
        if prompt_tokens > self.max_prompt_tokens:
            logger.warning(f"Prompt too long ({prompt_tokens} tokens), reducing context...")
            # Further reduce context
            optimized_context = self.truncate_context(context, self.max_context_tokens // 2)
            prompt = self._create_enhanced_prompt(question, optimized_context)
            prompt_tokens = self.count_tokens(prompt)
            logger.info(f"Reduced prompt tokens: {prompt_tokens}")
        
        try:
            # Call OpenAI API with enhanced settings
            response = await self._call_openai_enhanced(prompt)
            
            # Extract and parse response
            answer_text = response['choices'][0]['message']['content']
            result = self._parse_enhanced_response(answer_text, question)
            
            # Add token usage
            result['token_usage'] = {
                'prompt_tokens': response['usage']['prompt_tokens'],
                'completion_tokens': response['usage']['completion_tokens'],
                'total_tokens': response['usage']['total_tokens'],
                'context_items_used': len(optimized_context)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error processing question: {str(e)}",
                'confidence': 0.1,
                'relevant_clauses': [],
                'explanation': 'Processing error occurred',
                'found_in_context': False,
                'token_usage': {'total_tokens': 0}
            }
    
    def _create_enhanced_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Create an enhanced prompt optimized for insurance document Q&A accuracy"""
        
        # Prepare context with better formatting
        context_text = self._format_context_for_accuracy(context)
        
        prompt = f"""You are an expert insurance policy analyst. Your task is to provide accurate, detailed answers about insurance policies based on the provided document context.

INSTRUCTIONS:
1. Read the context carefully and identify all relevant information under 70 words.
2. Provide specific, accurate answers with exact details (numbers, percentages, timeframes)
3. Use the exact terminology from the policy document
4. If multiple conditions apply, list them all
5. Be specific about waiting periods, limits, exclusions, and conditions

CONTEXT FROM INSURANCE POLICY:
{context_text}

QUESTION: {question}

REQUIREMENTS FOR YOUR ANSWER:
- Be comprehensive and detailed (50-150 words)
- Include specific numbers, percentages, or timeframes mentioned in the policy
- Use exact policy language when possible
- If there are conditions or exceptions, mention them
- If information is not in the context, state that clearly
- Structure your answer logically

Provide a detailed, accurate answer based solely on the policy context above:"""
        
        return prompt
    
    def _format_context_for_accuracy(self, context: List[Dict[str, Any]]) -> str:
        """Format context to maximize accuracy"""
        if not context:
            return "No relevant context found."
        
        context_parts = []
        
        for i, item in enumerate(context):
            content = item['content'].strip()
            score = item.get('score', 0)
            
            # Add relevance indicator
            relevance = "HIGH" if score > 0.8 else "MEDIUM" if score > 0.6 else "LOW"
            
            context_parts.append(f"SECTION {i+1} (Relevance: {relevance}):\n{content}")
            
            # Add source info if available
            metadata = item.get('metadata', {})
            if metadata.get('filename'):
                context_parts.append(f"Source: {metadata['filename']}\n")
        
        return "\n\n".join(context_parts)
    
    async def _call_openai_enhanced(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API with enhanced settings for accuracy"""
        import httpx
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert insurance policy analyst who provides accurate, detailed answers based on policy documents. Always be specific with numbers, dates, conditions, and policy terms."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.1,  # More focused responses
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()
    
    def _parse_enhanced_response(self, response_text: str, question: str) -> Dict[str, Any]:
        """Parse the enhanced response from LLM"""
        
        # Clean up the response
        answer = response_text.strip()
        
        # Calculate confidence based on answer quality
        confidence = self._calculate_confidence(answer, question)
        
        # Extract any specific clauses or conditions mentioned
        relevant_clauses = self._extract_clauses(answer)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'relevant_clauses': relevant_clauses,
            'explanation': 'Answer generated using enhanced prompting for insurance policy analysis',
            'found_in_context': len(answer) > 20 and "not found" not in answer.lower() and "no information" not in answer.lower()
        }
    
    def _calculate_confidence(self, answer: str, question: str) -> float:
        """Calculate confidence score based on answer quality"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for specific details
        if any(char.isdigit() for char in answer):  # Contains numbers
            confidence += 0.2
        
        if any(word in answer.lower() for word in ['percent', '%', 'days', 'months', 'years', 'rupees', 'amount']):
            confidence += 0.15
        
        if any(word in answer.lower() for word in ['policy', 'coverage', 'benefit', 'condition', 'exclusion']):
            confidence += 0.1
        
        # Decrease confidence for vague answers
        if any(phrase in answer.lower() for phrase in ['not clear', 'unclear', 'not specified', 'not mentioned']):
            confidence -= 0.2
        
        # Decrease confidence for very short answers
        if len(answer.split()) < 10:
            confidence -= 0.15
        
        return max(0.1, min(1.0, confidence))
    
    def _extract_clauses(self, answer: str) -> List[str]:
        """Extract relevant policy clauses from the answer"""
        clauses = []
        
        # Look for specific policy terms
        clause_indicators = [
            'waiting period', 'grace period', 'coverage limit', 'exclusion',
            'condition', 'benefit', 'premium', 'deductible', 'co-payment'
        ]
        
        for indicator in clause_indicators:
            if indicator in answer.lower():
                clauses.append(indicator.title())
        
        return clauses

# Singleton instance
llm_service = LLMService()


