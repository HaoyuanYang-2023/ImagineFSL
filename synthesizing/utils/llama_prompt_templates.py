TEMPLATE_new = {
    "BG": """Your task is to generate an image caption for a {}, by considering the following factors: attribute, viewpoint, and background. The caption should be suitable for use as a textual prompt for Stable Diffusion, ensuring that the generated image resembles a real-life photo. Use the three examples provided below to guide the generation of the caption:
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} =>""",
    
    "LC": """Your task is to generate an image caption for a {}, by considering the following factors: attribute, viewpoint, and lighting conditions. The caption should be suitable for use as a textual prompt for Stable Diffusion, ensuring that the generated image resembles a real-life photo. Use the three examples provided below to guide the generation of the caption:
    {}, {} and {} => {}
    
    {}, {} and {} => {}

    {}, {} and {} => {}
    
    {}, {} and {} =>""",
    
    "CD": """Your task is to generate an image caption for a {}, by considering the following factors: attribute, viewpoint, and degradation causes resulting in deterioration of image quality. The caption should be suitable for use as a textual prompt for Stable Diffusion, ensuring that the generated image resembles a real-life photo. Use the three examples provided below to guide the generation of the caption:
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} =>""",
    
    "Base": """Your task is to generate an image caption for a {}, by considering the following factors: attribute and viewpoint. The caption should be suitable for use as a textual prompt for Stable Diffusion, ensuring that the generated image resembles a real-life photo. Use the three examples provided below to guide the generation of the caption:
    {}, {} => {}
    
    {}, {} => {}
    
    {}, {} => {}
    
    {}, {} =>"""
    
}