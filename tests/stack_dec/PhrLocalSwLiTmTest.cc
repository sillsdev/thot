#include "stack_dec/PhrLocalSwLiTm.h"

#include "incr_models/IncrJelMerNgramLM.h"
#include "incr_models/WordPenaltyModel.h"
#include "phrase_models/WbaIncrPhraseModel.h"
#include "stack_dec/TranslationMetadata.h"
#include "stack_dec/multi_stack_decoder_rec.h"
#include "sw_models/Ibm1AlignmentModel.h"

#include <gtest/gtest.h>
#include <memory>

class PhrLocalSwLiTmTest : public testing::Test
{
protected:
  PhrLocalSwLiTmTest()
  {
  }

  void createModel()
  {
    model.reset(new PhrLocalSwLiTm);

    auto langModelInfo = new LangModelInfo;
    auto phrModelInfo = new PhraseModelInfo;
    auto swModelInfo = new SwModelInfo;

    langModelInfo->wpModel.reset(new WordPenaltyModel);
    langModelInfo->langModel.reset(new IncrJelMerNgramLM);

    phrModelInfo->invPhraseModel.reset(new WbaIncrPhraseModel);
    swModelInfo->swAligModels.push_back(std::unique_ptr<Ibm1AlignmentModel>(new Ibm1AlignmentModel));
    swModelInfo->invSwAligModels.push_back(std::unique_ptr<Ibm1AlignmentModel>(new Ibm1AlignmentModel));

    model->setLangModelInfo(langModelInfo);
    model->setPhraseModelInfo(phrModelInfo);
    model->setSwModelInfo(swModelInfo);
    model->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);
  }

  void createDecoder()
  {
    decoder.reset(new multi_stack_decoder_rec<PhrLocalSwLiTm>);

    decoder->setParentSmtModel(model.get());
    auto smtModel = dynamic_cast<PhrLocalSwLiTm*>(model->clone());
    smtModel->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);
    decoder->setSmtModel(smtModel);
  }

  std::unique_ptr<PhrLocalSwLiTm> model;
  std::unique_ptr<multi_stack_decoder_rec<PhrLocalSwLiTm>> decoder;
};

TEST_F(PhrLocalSwLiTmTest, construct)
{
  createModel();

  EXPECT_EQ(model->getSwModelInfo()->swAligModels.size(), 1);
  EXPECT_EQ(model->getSwModelInfo()->invSwAligModels.size(), 1);
}

TEST_F(PhrLocalSwLiTmTest, constructDecoder)
{
  createModel();
  createDecoder();

  EXPECT_EQ(decoder->getParentSmtModel(), model.get());
  EXPECT_NE(decoder->getSmtModel(), model.get());
  EXPECT_EQ(decoder->getSmtModel()->getLangModelInfo(), model->getLangModelInfo());
  EXPECT_EQ(decoder->getSmtModel()->getPhraseModelInfo(), model->getPhraseModelInfo());
  EXPECT_EQ(decoder->getSmtModel()->getSwModelInfo(), model->getSwModelInfo());
  EXPECT_NE(decoder->getSmtModel()->getTranslationMetadata(), model->getTranslationMetadata());

  decoder.reset();

  EXPECT_EQ(model->getSwModelInfo()->swAligModels.size(), 1);
  EXPECT_EQ(model->getSwModelInfo()->invSwAligModels.size(), 1);
}
