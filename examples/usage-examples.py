"""
Examples of using KoboldAPI for various text, image, and video tasks.
"""

from pathlib import Path
from koboldapi import (
    KoboldAPICore,
    ImageProcessor,
    VideoProcessor,
    KoboldAPIConfig,
    ChunkingProcessor
)

def setup_core(api_url="http://localhost:5001"):
    """Setup basic KoboldAPICore instance."""
    config = {
        "api_url": api_url,
        "templates_directory": "./templates",
        "translation_language": "English",
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": 0,
        "rep_pen": 1.05
    }
    return KoboldAPICore(config)

# ==================
# Text Processing
# ==================

def translate_document(file_path: str, target_language: str = "French"):
    """Example: Translate a document to another language."""
    core = setup_core()
    chunker = ChunkingProcessor(core.api_client, max_chunk_length=1000)

    chunks, metadata = chunker.chunk_file(file_path)
    results = []
    
    for i, (chunk, _) in enumerate(chunks, 1):
        print(f"Translating chunk {i}/{len(chunks)}")
        
        prompt = core.template_wrapper.wrap_prompt(
            instruction=f"Translate the following text into {target_language}:",
            content=chunk,
            system_instruction=(
                "You are a skilled translator. Maintain the original "
                "style and tone while producing natural translations."
            )
        )
        
        translation = core.api_client.generate(
            prompt=prompt,
            max_length=len(chunk) * 2,  # Conservative estimate
            temperature=0.1,  # Lower temperature for translations
            top_p=1.0
        )
        results.append(translation)
    
    output_path = Path(file_path).with_suffix(f'.{target_language}.txt')
    output_path.write_text('\n\n'.join(results))
    print(f"Translation saved to: {output_path}")

def summarize_article(file_path: str, style: str = "academic"):
    """Example: Generate summaries in different styles."""
    core = setup_core()
    content = Path(file_path).read_text()
 
    style_configs = {
        "academic": {
            "instruction": (
                "Create a detailed academic summary of this text. Include main "
                "arguments, methodologies, and conclusions."
            ),
            "system": "You are an academic researcher skilled in analysis."
        },
        "business": {
            "instruction": (
                "Create an executive summary focusing on key business implications, "
                "actionable insights, and strategic recommendations."
            ),
            "system": "You are a business analyst producing executive summaries."
        },
        "simple": {
            "instruction": (
                "Summarize this text in simple, clear language that anyone can "
                "understand. Focus on the main points."
            ),
            "system": "You are skilled at making complex topics accessible."
        }
    }
    
    config = style_configs.get(style, style_configs["simple"])
    prompt = core.template_wrapper.wrap_prompt(
        instruction=config["instruction"],
        content=content,
        system_instruction=config["system"]
    )
    
    summary = core.api_client.generate(
        prompt=prompt,
        max_length=2000
    )
    
    output_path = Path(file_path).with_suffix(f'.summary.txt')
    output_path.write_text(summary)
    print(f"Summary saved to: {output_path}")

# ==================
# Image Processing
# ==================

def analyze_artwork(image_path: str):
    """Example: Detailed art analysis."""
    core = setup_core()
    processor = ImageProcessor(core)
    
    prompts = [
        "Analyze this artwork's composition, use of color, and technique.",
        "Identify the artistic style, period, and potential influences.",
        "Describe the emotional impact and symbolic elements.",
    ]
    
    analyses = {}
    for prompt in prompts:
        result, _ = processor.process_image(
            image_path,
            instruction=prompt,
            system_instruction=(
                "You are an art historian with expertise in "
                "multiple periods and styles."
            )
        )
        if result:
            analyses[prompt] = result
    
    # Save comprehensive analysis
    output_path = Path(image_path).with_suffix('.analysis.txt')
    with output_path.open('w') as f:
        for prompt, analysis in analyses.items():
            f.write(f"# {prompt}\n\n{analysis}\n\n")
    print(f"Art analysis saved to: {output_path}")

def batch_image_comparison(directory: str):
    """Example: Compare multiple images in a directory."""
    core = setup_core()
    processor = ImageProcessor(core)
    
    image_paths = list(Path(directory).glob("*.jpg"))
    if len(image_paths) < 2:
        print("Need at least 2 images for comparison")
        return
        
    results = processor.process_batch(
        image_paths,
        instruction=(
            "Compare and contrast this image with the others in terms of "
            "subject matter, style, composition, and emotional impact. "
            "Identify any themes or patterns."
        ),
        system_instruction=(
            "You are a visual analysis expert skilled at identifying "
            "patterns and relationships between images."
        ),
        output_dir=Path(directory) / "analysis"
    )
    
    # Generate overall comparison
    prompt = core.template_wrapper.wrap_prompt(
        instruction=(
            "Based on the individual analyses, provide a comprehensive "
            "comparison of all images, highlighting key patterns, "
            "contrasts, and shared elements."
        ),
        content="\n\n".join(
            f"Image {i+1}: {r['result']}"
            for i, r in enumerate(results.values())
            if isinstance(r, dict) and r.get('result')
        )
    )
    
    comparison = core.api_client.generate(prompt=prompt, max_length=2000)
    
    summary_path = Path(directory) / "comparison_summary.txt"
    summary_path.write_text(comparison)
    print(f"Comparison summary saved to: {summary_path}")

# ==================
# Video Processing
# ==================

def character_analysis(video_path: str):
    """Example: Analyze character appearances and interactions."""
    core = setup_core()
    processor = VideoProcessor(core)
    
    results = processor.analyze_video(
        video_path,
        max_frames=64,
        batch_size=8
    )
    
    # Focus on character analysis
    prompt = core.template_wrapper.wrap_prompt(
        instruction=(
            "Based on the scene descriptions, provide a detailed analysis of:\n"
            "1. Main characters and their distinguishing features\n"
            "2. Character interactions and relationships\n"
            "3. Character development or changes\n"
            "4. Recurring patterns in character behavior"
        ),
        content="\n\n".join(
            analysis["analysis"] for analysis in results["analysis"]
        )
    )
    
    character_study = core.api_client.generate(
        prompt=prompt,
        max_length=2000
    )
    
    # Save analysis
    output_dir = Path(video_path).parent / f"{Path(video_path).stem}_analysis"
    output_dir.mkdir(exist_ok=True)
    
    analysis_path = output_dir / "character_analysis.txt"
    analysis_path.write_text(character_study)
    print(f"Character analysis saved to: {analysis_path}")

def scene_detection(video_path: str):
    """Example: Detect and analyze scene transitions."""
    core = setup_core()
    processor = VideoProcessor(core)
    
    results = processor.analyze_video(
        video_path,
        max_frames=128,
        batch_size=8
    )
    
    # Analyze scene transitions
    prompt = core.template_wrapper.wrap_prompt(
        instruction=(
            "Based on the frame descriptions, identify and analyze:\n"
            "1. Major scene transitions\n"
            "2. Visual themes and continuity between scenes\n"
            "3. Pacing and rhythm of scene changes\n"
            "4. Notable cinematographic techniques"
        ),
        content="\n\n".join(
            analysis["analysis"] for analysis in results["analysis"]
        )
    )
    
    scene_analysis = core.api_client.generate(
        prompt=prompt,
        max_length=2000
    )
    
    output_dir = Path(video_path).parent / f"{Path(video_path).stem}_analysis"
    output_dir.mkdir(exist_ok=True)
    
    analysis_path = output_dir / "scene_analysis.txt"
    analysis_path.write_text(scene_analysis)
    print(f"Scene analysis saved to: {analysis_path}")


if __name__ == "__main__":
    translate_document("article.pdf", "Spanish")
    summarize_article("article.txt", "academic")
    analyze_artwork("painting.jpg")
    batch_image_comparison("image_directory")
    character_analysis("movie.mp4")
    scene_detection("movie.mp4")