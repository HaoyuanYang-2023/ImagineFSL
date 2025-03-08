TEMPLATE_new = {
    "BG": """You will generate an image caption for {} by considering the following factors: attributes, shooting angles/distances and background. The caption should be generated in such a way that, when used as a text prompt for stable diffusion, the generated images should be similar to real-photos photographed in everyday life. You will be given three examples as follows:
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} =>""",
    
    "LC": """You will generate an image caption for a {} by considering the following factors: attributes, shooting angles/distances and lighting conditions. The caption should be generated in such a way that, when used as a text prompt for stable diffusion, the generated images should be similar to real-photos photographed in everyday life. You will be given three examples as follows:
    
    {}, {} and {} => {}
    
    {}, {} and {} => {}

    {}, {} and {} => {}
    
    {}, {} and {} =>""",
    
    "CD": """You will generate an image caption for a {} by considering the following factors: attributes, shooting angles/distances and degradation causes resulting in deterioration of image quality. The caption should be generated in such a way that, when used as a text prompt for stable diffusion, the generated images should be similar to real-photos photographed in everyday life. You will be given three examples as follows:
    
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} => {}
    
    {}, {} and {} =>""",
    
    "Base": """You will generate an image caption for {} by considering the following factors: attribute and shooting angles/distance. The caption should be generated in such a way that, when used as a text prompt for stable diffusion, the generated images should be similar to real-photos photographed in everyday life. You will be given three examples as follows:
    
    {}, {} => {}
    
    {}, {} => {}
    
    {}, {} => {}
    
    {}, {} =>"""
    
}