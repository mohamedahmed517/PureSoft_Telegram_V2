"""
Gemini AI chat service
"""
import io
import time
import base64
from PIL import Image
from datetime import datetime
from google.genai import types
from utils.logger import logger
from utils.metrics import metrics
from web_database import save_web_conversation
from services.products import build_product_catalog
from models import CLIENT, GENERATION_CONFIG, SAFETY_SETTINGS
from services.history import conversation_history, get_conversation_context, add_message

def gemini_chat(text="", image_b64=None, audio_data=None, user_key="unknown"):
    """Main chat function with Gemini AI"""
    start_time = time.time()
    max_retries = 2
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        history_text, recent_messages = get_conversation_context(user_key)
        products_text = build_product_catalog()
        prompt = f"""
أنت البوت الذكي بتاع آفاق ستورز، بتتكلم عامية مصرية ودودة وطبيعية.
أنت مساعد شامل بتعرف تتكلم في أي موضوع: موضة، سياحة، طقس، أماكن خروج، نصايح عامة، وأي حاجة تانية.

آخر محادثة:
{history_text}

دول كل المنتجات اللي موجودة عندنا دلوقتي (خد بالك من الأسماء دي بالحرف لأن اللينكات مربوطة بيها):
{products_text}

آخر رسايل المحادثة:
{recent_messages}

العميل بيقول دلوقتي: {text or "بعت صورة" if not audio_data else "بعت صوت"}

**قواعد الرد:**

1. **لو العميل سألك عن حاجة عامة** → رد عليه عادي وطبيعي من غير ما ترشح منتجات إلا لو هو طلب كده بوضوح.

2. **لو سأل عن الطقس في مكان معين** → 
   - قوله الطقس إيه بطريقة ودودة.
   - قدم نصيحة عامة للبس المناسب.
   - **اقترح 2-6 منتجات متنوعة تناسب الجو ده**، مش مجرد لبس.
   - **تنسيق عرض المنتجات:** استخدم التنسيق ده بالضبط لكل منتج:

اسم المنتج بالضبط
السعر: [السعر] جنيه
الكاتيجوري: [الفئة]
اللينك: [الرابط]

(سطر فاضي بين كل منتج واللي بعده)

3. **لو سأل عن أماكن خروج أو فسح** → 
   - رشحله أماكن حلوة واديله معلومات مفيدة.
   - **اقترح 2-6 منتجات تناسب الخروجات**.

4. **لو قالك "مترشحليش منتجات دلوقتي" أو "مش عايز أشوف منتجات"** → مترشحش خالص واتكلم عادي.

5. **لو طلب منك صراحة يشوف منتجات معينة** → رشحله من المنتجات اللي فوق.

6. **نظام الكلمات المفتاحية الشامل:**
   لو العميل استخدم أي من الكلمات دي في كلامه **حتى لو مش طلب صريح**، ابدأ اقتراح **2-6 منتجات متكاملة** تناسب الموقف.

   **الكلمات المفتاحية:**
   - برد / شتا / شتوية / جو بارد / مطر
   - حر / صيف / جو حار / شمس
   - ربيع / خريف / جو معتدل
   - سهرة / حفلة / أفراح / عزومة / مناسبة خاصة
   - شغل / مكتب / وظيفة / اجتماع
   - جامعة / دراسة / كليه
   - رياضة / جيم / تمرين
   - سفر / رحلة / أجازة / رحلات
   - بيت / منزل / أثاث / ديكور
   - هدية / تخرج / عيد ميلاد
   - تخفيضات / عروض / رخيص
   - جديد / تشكيلة جديدة

   **طريقة الاقتراح:**
   - ابدأ بقول: "بالنسبة لـ [الكلمة المفتاحية/الموضوع]، عندنا حاجات كويسة ممكن تفيدك:"
   - قدم المنتجات بالتنسيق المطلوب بالضبط:

اسم المنتج بالضبط
السعر: [السعر] جنيه
الكاتيجوري: [الفئة]
اللينك: [الرابط]

7. **مثال توضيحي للتنسيق المطلوب:**
   هذا هو التنسيق الصحيح الذي يجب استخدامه عند عرض أي منتج:

   تيشيرت قطن سادة ابيض
   السعر: 130 جنيه
   الكاتيجوري: لبس صيفي
   اللينك: https://afaq-stores.com/product-details/1019

   سكارف كشمير طويل
   السعر: 290 جنيه
   الكاتيجوري: لبس خريفي
   اللينك: https://afaq-stores.com/product-details/1014

   جاكيت جلد اسود تقيل مبطن فرو
   السعر: 720 جنيه
   الكاتيجوري: لبس شتوي
   اللينك: https://afaq-stores.com/product-details/1001

   **ممنوع تماماً:** استخدام النقاط (*) أو الشرطات (-) أو الأقواس (**) أو الأنابيب (|) في عرض المنتجات.

8. **لو بعت صورة منتج** → ابدأ الرد بـ "ثانية بس أشوف الصورة..." وقدم بدائل مشابهة.
9. **لو بعت صوت** → اسمع الصوت كويس ورد على اللي بيقوله مع تطبيق نظام الكلمات المفتاحية.

**أسلوب الكلام:**
- اتكلم عامية مصرية 100% وبشكل طبيعي.
- متحطش إيموجي خالص.
- متقولش إنك بوت أبدًا.
- كن مفيد وودود في أي موضوع.
- لو مش عارف حاجة، اعترف بكده بشكل طبيعي.
- الرد الطبيعي يكون: أولاً الرد على سؤال العميل، ثانياً اقتراح المنتجات إذا كان مناسباً، ثالثاً سؤال "تحب أساعدك في أي حاجة تانية؟"

**هام جداً:** عندما تعرض منتجات، تأكد من استخدام التنسيق المطلوب فقط (كل سطر منفصل) بدون أي رموز أو علامات إضافية.

رد دلوقتي:
""".strip()
        response = None
        for attempt in range(max_retries):
            try:
                config_dict = GENERATION_CONFIG.model_dump() if hasattr(GENERATION_CONFIG, 'model_dump') else GENERATION_CONFIG.dict()
                config_dict.pop('safety_settings', None)
                
                if audio_data:
                    response = CLIENT.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            prompt,
                            {"mime_type": "audio/ogg", "data": audio_data}
                        ],
                        config=types.GenerateContentConfig(
                            **config_dict,
                            safety_settings=SAFETY_SETTINGS
                        )
                    )
                    metrics.track_message("with_audio")
                   
                elif image_b64:
                    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                   
                    response = CLIENT.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            prompt,
                            {"mime_type": "image/png", "data": img_bytes.read()}
                        ],
                        config=types.GenerateContentConfig(
                            **config_dict,
                            safety_settings=SAFETY_SETTINGS
                        )
                    )
                    metrics.track_message("with_image")
                   
                else:
                    response = CLIENT.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            **config_dict,
                            safety_settings=SAFETY_SETTINGS
                        )
                    )
                    metrics.track_message("text_only")
                   
                break
               
            except Exception as e:
                logger.warning(f"⚠️ Gemini API attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise
       
        reply = response.text.strip() if response and hasattr(response, "text") and response.text else "ثواني بس فيه مشكلة دلوقتي..."
        
        add_message(user_key, "user", text or ("[صورة]" if image_b64 else "[صوت]"), now)
        add_message(user_key, "assistant", reply, now)
        
        if user_key.startswith("web:"):
            try:
                user_id = int(user_key.split(":")[1])
                history = conversation_history.get(user_key, [])
                save_web_conversation(user_id, history)
                logger.info(f"💾 Saved web conversation for user {user_id}")
            except Exception as e:
                logger.error(f"❌ Error saving web conversation: {e}")
        
        response_time = time.time() - start_time
        metrics.track_response_time(response_time)
       
        logger.info(f"✅ Response generated for {user_key} in {response_time:.2f}s")
        return reply
    except Exception as e:
        logger.error(f"❌ Error in gemini_chat: {e}", exc_info=True)
        metrics.track_error("gemini_chat")
        return "ثواني بس فيه مشكلة دلوقتي هحلها وارجعلك..."


